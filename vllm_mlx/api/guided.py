# SPDX-License-Identifier: Apache-2.0
"""
Guided generation for structured JSON output using outlines.

This module provides constrained decoding for JSON schema enforcement,
ensuring model outputs strictly adhere to specified schemas.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Check for outlines availability
try:
    import mlx_lm
    import outlines

    HAS_OUTLINES = True
except ImportError:
    HAS_OUTLINES = False
    outlines = None
    mlx_lm = None


def is_guided_available() -> bool:
    """Check if guided generation with outlines is available."""
    return HAS_OUTLINES


def json_schema_to_pydantic(schema: dict[str, Any]) -> type | None:
    """
    Convert a JSON schema to a Pydantic model dynamically.

    Args:
        schema: JSON schema dict

    Returns:
        Dynamically created Pydantic model class, or None if conversion fails
    """
    try:
        from pydantic import create_model

        # Extract properties from schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Build field definitions for Pydantic
        field_definitions = {}

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "null": type(None),
        }

        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")

            # Handle array type
            if prop_type == "array":
                items_type = prop_spec.get("items", {}).get("type", "string")
                inner_type = type_mapping.get(items_type, str)
                python_type = list[inner_type]
            # Handle object type (nested)
            elif prop_type == "object":
                # For nested objects, use dict
                python_type = dict
            else:
                python_type = type_mapping.get(prop_type, str)

            # Make optional if not required
            if prop_name not in required:
                python_type = python_type | None
                default = None
            else:
                default = ...

            field_definitions[prop_name] = (python_type, default)

        # Create the model dynamically
        model = create_model("DynamicModel", **field_definitions)
        return model

    except Exception as e:
        logger.warning(f"Failed to convert JSON schema to Pydantic: {e}")
        logger.debug(f"Problematic schema: {schema}")
        return None


class GuidedGenerator:
    """
    Guided generation using outlines for constrained JSON decoding.

    This class wraps an MLX model to provide structured output generation
    that guarantees valid JSON matching a specified schema.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize the guided generator.

        Args:
            model: MLX model instance
            tokenizer: Tokenizer instance
        """
        if not HAS_OUTLINES:
            raise ImportError(
                "outlines is required for guided generation. "
                "Install with: pip install 'vllm-mlx[guided]'"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._outlines_model = None

    def _get_outlines_model(self):
        """Get or create the outlines model wrapper."""
        if self._outlines_model is None:
            self._outlines_model = outlines.from_mlxlm(self._model, self._tokenizer)
        return self._outlines_model

    def generate_json(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate JSON output constrained to a schema.

        Args:
            prompt: Input prompt
            json_schema: JSON schema to constrain output
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            JSON string matching the schema
        """
        # Convert schema to Pydantic model
        pydantic_model = json_schema_to_pydantic(json_schema)

        if pydantic_model is None:
            logger.warning(
                "Could not convert schema to Pydantic, falling back to raw generation"
            )
            return None

        try:
            outlines_model = self._get_outlines_model()

            # Generate with schema constraint
            result = outlines_model(
                prompt,
                output_type=pydantic_model,
                max_tokens=max_tokens,
            )

            # result is a JSON string, validate and return
            return result

        except Exception as e:
            logger.error(f"Guided generation failed: {e}")
            return None

    def generate_json_object(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate any valid JSON object.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            JSON string
        """
        try:
            from outlines import generate

            outlines_model = self._get_outlines_model()

            # Use regex to constrain to valid JSON
            json_regex = r"\{[^{}]*\}"
            generator = generate.regex(outlines_model, json_regex)
            result = generator(prompt, max_tokens=max_tokens)

            return result

        except Exception as e:
            logger.error(f"JSON object generation failed: {e}")
            return None


def generate_with_schema(
    model,
    tokenizer,
    prompt: str,
    json_schema: dict[str, Any],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """
    Convenience function for one-shot guided JSON generation.

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        json_schema: JSON schema
        max_tokens: Maximum tokens
        temperature: Sampling temperature

    Returns:
        JSON string or None if guided generation unavailable/failed
    """
    if not HAS_OUTLINES:
        return None

    try:
        generator = GuidedGenerator(model, tokenizer)
        return generator.generate_json(
            prompt=prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"generate_with_schema failed: {e}")
        return None
