
# SPDX-License-Identifier: Apache-2.0
"""Tests for guided generation module."""

import pytest
from unittest.mock import MagicMock, patch


class TestJsonSchemaToPydantic:
    """Tests for json_schema_to_pydantic function."""

    def test_basic_string_property(self):
        """Test conversion of basic string property."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        assert hasattr(model, "model_validate")
        
        # Test validation
        instance = model.model_validate({"name": "test"})
        assert instance.name == "test"

    def test_multiple_property_types(self):
        """Test conversion of multiple property types."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"}
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({
            "name": "John",
            "age": 30,
            "score": 85.5,
            "active": True
        })
        assert instance.name == "John"
        assert instance.age == 30
        assert instance.score == 85.5
        assert instance.active is True

    def test_required_fields(self):
        """Test required fields are enforced."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name"]
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        # Should fail without required field
        with pytest.raises(Exception):
            model.model_validate({})
        
        # Should pass with required field
        instance = model.model_validate({"name": "test"})
        assert instance.name == "test"
        assert instance.email is None  # Optional field

    def test_optional_fields(self):
        """Test optional fields work correctly."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "optional_field": {"type": "string"}
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({})
        assert instance.optional_field is None

    def test_array_type(self):
        """Test array type conversion."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({
            "tags": ["a", "b", "c"],
            "scores": [1.0, 2.5, 3.5]
        })
        assert instance.tags == ["a", "b", "c"]
        assert instance.scores == [1.0, 2.5, 3.5]

    def test_nested_object(self):
        """Test nested object type conversion."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({
            "user": {"name": "John", "age": 30}
        })
        assert instance.user == {"name": "John", "age": 30}

    def test_empty_schema(self):
        """Test empty schema returns model with no fields."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {"type": "object", "properties": {}}
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({})
        assert hasattr(instance, "model_validate")

    def test_missing_properties(self):
        """Test schema without properties key."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {"type": "object"}
        
        model = json_schema_to_pydantic(schema)
        assert model is not None

    def test_complex_schema(self):
        """Test complex schema with mixed types."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "is_active": {"type": "boolean"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string"}
                    }
                }
            },
            "required": ["id", "count"]
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({
            "id": "abc123",
            "count": 5,
            "price": 10.99,
            "is_active": True,
            "items": ["a", "b"],
            "metadata": {"created": "2024-01-01"}
        })
        assert instance.id == "abc123"
        assert instance.count == 5


class TestIsGuidedAvailable:
    """Tests for is_guided_available function."""

    @patch("vllm_mlx.api.guided.HAS_OUTLINES", True)
    def test_returns_true_when_outlines_available(self):
        """Test returns True when outlines is available."""
        from vllm_mlx.api.guided import is_guided_available
        
        # Need to reimport to pick up the patch
        import vllm_mlx.api.guided as guided
        result = guided.is_guided_available()
        assert result is True

    @patch("vllm_mlx.api.guided.HAS_OUTLINES", False)
    def test_returns_false_when_outlines_not_available(self):
        """Test returns False when outlines is not available."""
        import vllm_mlx.api.guided as guided
        result = guided.is_guided_available()
        assert result is False


class TestGuidedGenerator:
    """Tests for GuidedGenerator class."""

    def test_init_raises_import_error_without_outlines(self):
        """Test that init raises ImportError when outlines not available."""
        import vllm_mlx.api.guided as guided
        
        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        original_outlines = guided.outlines
        
        try:
            guided.HAS_OUTLINES = False
            guided.outlines = None
            
            with pytest.raises(ImportError, match="outlines is required"):
                guided.GuidedGenerator(None, None)
        finally:
            # Restore original state
            guided.HAS_OUTLINES = original_has_outlines
            guided.outlines = original_outlines

    def test_init_succeeds_with_outlines(self):
        """Test init succeeds when outlines is available."""
        import vllm_mlx.api.guided as guided
        
        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        original_outlines = guided.outlines
        
        try:
            guided.HAS_OUTLINES = True
            mock_outlines = MagicMock()
            guided.outlines = mock_outlines
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            
            generator = guided.GuidedGenerator(mock_model, mock_tokenizer)
            assert generator._model is mock_model
            assert generator._tokenizer is mock_tokenizer
        finally:
            guided.HAS_OUTLINES = original_has_outlines
            guided.outlines = original_outlines

    def test_generate_json_with_mocked_outlines(self):
        """Test generate_json with mocked outlines."""
        import vllm_mlx.api.guided as guided
        
        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        original_outlines = guided.outlines
        
        try:
            guided.HAS_OUTLINES = True
            
            # Create mock outlines
            mock_outlines = MagicMock()
            guided.outlines = mock_outlines
            
            # Create mock model and tokenizer
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            
            # Create mock outlines model that returns JSON
            mock_outlines_model = MagicMock()
            mock_outlines.from_mlxlm.return_value = mock_outlines_model
            
            # Create generator
            generator = guided.GuidedGenerator(mock_model, mock_tokenizer)
            
            # Test schema
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
            
            # Mock the result
            mock_outlines_model.return_value = '{"name": "test"}'
            
            result = generator.generate_json(
                prompt="Generate a name",
                json_schema=schema,
                max_tokens=100,
                temperature=0.5
            )
            
            assert result == '{"name": "test"}'
            mock_outlines.from_mlxlm.assert_called_once_with(mock_model, mock_tokenizer)
        finally:
            guided.HAS_OUTLINES = original_has_outlines
            guided.outlines = original_outlines

    def test_generate_json_returns_none_on_failure(self):
        """Test generate_json returns None on failure."""
        import vllm_mlx.api.guided as guided
        
        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        original_outlines = guided.outlines
        
        try:
            guided.HAS_OUTLINES = True
            
            # Create mock outlines that raises exception
            mock_outlines = MagicMock()
            guided.outlines = mock_outlines
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_outlines_model = MagicMock()
            mock_outlines.from_mlxlm.return_value = mock_outlines_model
            
            # Make the model raise an exception
            mock_outlines_model.side_effect = Exception("Generation failed")
            
            generator = guided.GuidedGenerator(mock_model, mock_tokenizer)
            
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
            
            result = generator.generate_json(
                prompt="Generate",
                json_schema=schema
            )
            
            assert result is None
        finally:
            guided.HAS_OUTLINES = original_has_outlines
            guided.outlines = original_outlines

    def test_generate_json_object_with_mocked_outlines(self):
        """Test generate_json_object with mocked outlines."""
        import sys
        import vllm_mlx.api.guided as guided

        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        original_outlines = guided.outlines
        had_outlines = "outlines" in sys.modules
        original_module = sys.modules.get("outlines")

        try:
            guided.HAS_OUTLINES = True

            # Create mock outlines module with generate submodule
            mock_outlines = MagicMock()
            mock_generate = MagicMock()
            mock_generator = MagicMock()
            mock_generate.regex.return_value = mock_generator
            mock_generator.return_value = '{"key": "value"}'

            mock_outlines.generate = mock_generate
            guided.outlines = mock_outlines
            # Also patch sys.modules so `from outlines import generate` works
            sys.modules["outlines"] = mock_outlines
            sys.modules["outlines.generate"] = mock_generate

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_outlines_model = MagicMock()
            mock_outlines.from_mlxlm.return_value = mock_outlines_model

            generator = guided.GuidedGenerator(mock_model, mock_tokenizer)

            result = generator.generate_json_object(
                prompt="Generate JSON",
                max_tokens=100,
                temperature=0.5
            )

            assert result == '{"key": "value"}'
            mock_generate.regex.assert_called_once()
        finally:
            guided.HAS_OUTLINES = original_has_outlines
            guided.outlines = original_outlines
            if had_outlines:
                sys.modules["outlines"] = original_module
            else:
                sys.modules.pop("outlines", None)
            sys.modules.pop("outlines.generate", None)


class TestGenerateWithSchema:
    """Tests for generate_with_schema convenience function."""

    def test_returns_none_when_outlines_not_available(self):
        """Test returns None when outlines is not available."""
        import vllm_mlx.api.guided as guided
        
        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        
        try:
            guided.HAS_OUTLINES = False
            
            result = guided.generate_with_schema(
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt="Generate",
                json_schema={"type": "object", "properties": {}}
            )
            
            assert result is None
        finally:
            guided.HAS_OUTLINES = original_has_outlines

    def test_generates_json_with_schema(self):
        """Test generate_with_schema produces JSON output."""
        import vllm_mlx.api.guided as guided
        
        # Save original state
        original_has_outlines = guided.HAS_OUTLINES
        original_outlines = guided.outlines
        
        try:
            guided.HAS_OUTLINES = True
            
            mock_outlines = MagicMock()
            guided.outlines = mock_outlines
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_outlines_model = MagicMock()
            mock_outlines.from_mlxlm.return_value = mock_outlines_model
            
            schema = {
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            }
            
            mock_outlines_model.return_value = '{"result": "success"}'
            
            result = guided.generate_with_schema(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="Generate",
                json_schema=schema,
                max_tokens=50,
                temperature=0.3
            )
            
            assert result == '{"result": "success"}'
        finally:
            guided.HAS_OUTLINES = original_has_outlines
            guided.outlines = original_outlines


class TestEdgeCases:
    """Test edge cases for the guided generation module."""

    def test_empty_schema_with_required(self):
        """Test empty schema with required fields."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        instance = model.model_validate({})
        assert instance is not None

    def test_schema_with_all_optional_fields(self):
        """Test schema where all fields are optional."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"}
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        # Should validate with empty dict
        instance = model.model_validate({})
        assert instance.field1 is None
        assert instance.field2 is None

    def test_array_with_integer_items(self):
        """Test array with integer items."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({"ids": [1, 2, 3]})
        assert instance.ids == [1, 2, 3]

    def test_array_with_boolean_items(self):
        """Test array with boolean items."""
        from vllm_mlx.api.guided import json_schema_to_pydantic
        
        schema = {
            "type": "object",
            "properties": {
                "flags": {
                    "type": "array",
                    "items": {"type": "boolean"}
                }
            }
        }
        
        model = json_schema_to_pydantic(schema)
        assert model is not None
        
        instance = model.model_validate({"flags": [True, False, True]})
        assert instance.flags == [True, False, True]

