# SPDX-License-Identifier: Apache-2.0
"""
Guided generation for structured JSON output using outlines.

This module provides constrained decoding for JSON schema enforcement,
ensuring model outputs strictly adhere to specified schemas.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# MUST install the MLX hardware-compat shim BEFORE the `mlx_lm` import below.
# Even though the import is inside a `try`, the body still runs at module
# load time; on success it triggers `mlx_lm/__init__.py` → `mlx_lm.generate`
# → `mx.new_thread_local_stream(...)` capture, which on M5 single-stream
# GPUs would be unusable (#404). The shim is idempotent and a no-op on
# hardware where the original API works.
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

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

            # Handle array type. The "object" and "array" element types
            # are special-cased: without this branch they fell through to
            # ``type_mapping.get(items_type, str)`` and silently became
            # ``list[str]``, so the model emitted strings where the schema
            # required objects — producing JSON that fails validation
            # against the user's own schema (R10 sweep, guided.py bug).
            if prop_type == "array":
                items_type = prop_spec.get("items", {}).get("type", "string")
                if items_type == "object":
                    python_type = list[dict]
                elif items_type == "array":
                    python_type = list[list]
                else:
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
                "Install with: pip install 'rapid-mlx[guided]'"
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
    ) -> str | None:
        """Generate JSON output constrained to a schema.

        Hands the raw schema dict to outlines via
        ``outlines.types.dsl.JsonSchema``, which natively understands
        ``$defs``, ``$ref``, ``anyOf``, ``enum``, numeric bounds,
        ``additionalProperties: false``, and nested objects. The
        previous code path passed the schema through
        ``json_schema_to_pydantic`` first — that converter silently
        dropped every one of those constructs, so outlines was given a
        Pydantic model that was a strict superset of the user's schema.
        On a real-world schema with ``$defs`` + ``$ref`` (waybarrios#546
        repro), this surfaced as outlines streaming a valid JSON array
        when the schema required an object. Pass the dict through and
        let outlines own schema interpretation.

        (We import the ``JsonSchema`` class directly rather than going
        through the top-level ``outlines.json_schema`` factory; see the
        in-function comment for why.)
        """
        try:
            outlines_model = self._get_outlines_model()

            # Use the ``JsonSchema`` class from ``outlines.types.dsl``
            # directly rather than the top-level ``outlines.json_schema``
            # factory. The factory is a convenience export — it landed
            # mid-way through the 1.x line and is absent on the floor of
            # our declared ``outlines>=1.0.0`` dependency range, where
            # this attribute lookup raises ``AttributeError`` (codex R5
            # P1: that exception silently bubbles to the catch below,
            # returns None, and ``generate_with_schema`` falls back to
            # *unconstrained* generation for every json_schema request
            # while the chat route logs "Using guided generation"). The
            # underlying ``JsonSchema`` class has been stable since the
            # feature first shipped, so importing it directly avoids
            # the surface-version dependency.
            #
            # Import failures are surfaced loudly (WARNING-level
            # logger.exception with full traceback) before returning
            # ``None`` so operators can detect that guided generation
            # was silently disabled — DeepSeek R2 found that the prior
            # bare ``logger.error`` swallowed the traceback for the new
            # ``outlines.types.dsl`` import path, which on an older
            # outlines without that submodule would otherwise look
            # indistinguishable from a runtime generation failure.
            from outlines.types.dsl import JsonSchema

            schema_constraint = JsonSchema(json_schema)
            result = outlines_model(
                prompt,
                output_type=schema_constraint,
                max_tokens=max_tokens,
            )
            return result

        except ImportError:
            # Specifically distinguish "outlines installed but
            # ``outlines.types.dsl`` missing/renamed" from a generic
            # runtime failure — this is the failure mode the comment
            # block above warns about. ``logger.exception`` includes
            # the full traceback so the operator sees the import path
            # that broke, not just a flat string.
            logger.exception(
                "Guided generation unavailable: import of outlines "
                "constraint API failed. Falling back to unconstrained "
                "generation."
            )
            return None
        except Exception:
            logger.exception("Guided generation failed")
            return None

    def generate_json_object(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str | None:
        """
        Generate any valid JSON object.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            JSON string, or None on failure
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
